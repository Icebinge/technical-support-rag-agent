from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bm25_k1_b_grid import (
    run_primeqa_hybrid_bm25_k1_b_grid,
    write_primeqa_hybrid_bm25_k1_b_grid_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage82 train/dev-only PrimeQA hybrid BM25 k1/b small grid."
)


@app.command()
def main(
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train split JSONL."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev split JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    stage75_report: Annotated[
        Path | None,
        typer.Option("--stage75-report", help="Stage75 BM25 baseline report JSON."),
    ] = None,
    stage76_report: Annotated[
        Path | None,
        typer.Option("--stage76-report", help="Stage76 candidate design report JSON."),
    ] = None,
    stage81_report: Annotated[
        Path | None,
        typer.Option("--stage81-report", help="Stage81 dense+sparse RRF report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage82 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage82 BM25 k1/b small grid and write a public-safe report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    train_split_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    stage75_report_path = stage75_report or (
        settings.artifact_dir / "primeqa_hybrid_bm25_top10_miss_analysis_stage75.json"
    )
    stage76_report_path = stage76_report or (
        settings.artifact_dir / "primeqa_hybrid_retrieval_recall_candidate_design_stage76.json"
    )
    stage81_report_path = stage81_report or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_bm25_k1_b_grid_stage82.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_bm25_k1_b_grid_stage82_visuals"
    )

    report = run_primeqa_hybrid_bm25_k1_b_grid(
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        stage75_report_path=stage75_report_path,
        stage76_report_path=stage76_report_path,
        stage81_report_path=stage81_report_path,
    )
    visualizations = write_primeqa_hybrid_bm25_k1_b_grid_visualizations(
        report=report,
        output_dir=visualization_output_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid BM25 k1/b grid report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "config": report["config"],
        "grid_configs": report["grid_configs"],
        "metrics_by_split": report["metrics_by_split"],
        "comparisons_to_baseline": report["comparisons_to_baseline"],
        "train_selection": report["train_selection"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
