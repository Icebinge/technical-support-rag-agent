from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_query_view_ablation import (
    run_primeqa_hybrid_query_view_ablation,
    write_primeqa_hybrid_query_view_ablation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage77 train/dev-only PrimeQA hybrid BM25 query-view ablation."
)


@app.command()
def main(
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL path."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev JSONL path."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    stage75_report: Annotated[
        Path | None,
        typer.Option("--stage75-report", help="Stage75 BM25 top10 miss report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage77 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="Comma-separated top-k values; must include 10."),
    ] = "1,5,10",
    bm25_k1: Annotated[
        float,
        typer.Option("--bm25-k1", help="BM25 k1 parameter."),
    ] = 1.5,
    bm25_b: Annotated[
        float,
        typer.Option("--bm25-b", help="BM25 b parameter."),
    ] = 0.75,
) -> None:
    """Run Stage77 query-view ablation and write the report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    train_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    stage75_report_path = stage75_report or (
        settings.artifact_dir / "primeqa_hybrid_bm25_top10_miss_analysis_stage75.json"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_query_view_ablation_stage77.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_query_view_ablation_stage77_visuals"
    )

    report = run_primeqa_hybrid_query_view_ablation(
        train_split_path=train_path,
        dev_split_path=dev_path,
        documents_path=documents_path,
        stage75_report_path=stage75_report_path,
        top_k_values=_parse_top_k_values(top_k),
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    visualizations = write_primeqa_hybrid_query_view_ablation_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid query-view ablation report: {output_path}")


def _parse_top_k_values(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise typer.BadParameter("--top-k must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise typer.BadParameter("--top-k values must be positive integers.")
    if 10 not in values:
        raise typer.BadParameter("--top-k must include 10 for Stage77.")
    return values


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "query_views": report["query_views"],
        "train_selection": report["train_selection"],
        "metrics_by_split": report["metrics_by_split"],
        "comparisons_to_baseline": report["comparisons_to_baseline"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
