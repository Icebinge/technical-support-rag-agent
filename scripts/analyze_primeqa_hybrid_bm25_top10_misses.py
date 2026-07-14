from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bm25_miss_analysis import (
    run_primeqa_hybrid_bm25_miss_analysis,
    write_primeqa_hybrid_bm25_miss_analysis_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage75 train/dev-only BM25 top10 miss analysis on PrimeQA hybrid data."
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
    candidate_dataset: Annotated[
        Path | None,
        typer.Option("--candidate-dataset", help="Stage69 train/dev candidate JSONL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage75 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Miss threshold retrieval depth."),
    ] = 10,
    search_depth: Annotated[
        int,
        typer.Option("--search-depth", help="Diagnostic rank depth for missed gold docs."),
    ] = 50,
    bm25_k1: Annotated[
        float,
        typer.Option("--bm25-k1", help="BM25 k1 parameter."),
    ] = 1.5,
    bm25_b: Annotated[
        float,
        typer.Option("--bm25-b", help="BM25 b parameter."),
    ] = 0.75,
) -> None:
    """Run Stage75 BM25 top10 miss analysis and write the report."""

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
    candidate_dataset_path = candidate_dataset or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.jsonl"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_bm25_top10_miss_analysis_stage75.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals"
    )

    for path in [train_path, dev_path, documents_path, candidate_dataset_path]:
        _ensure_file_exists(path)

    report = run_primeqa_hybrid_bm25_miss_analysis(
        train_split_path=train_path,
        dev_split_path=dev_path,
        documents_path=documents_path,
        candidate_dataset_path=candidate_dataset_path,
        top_k=top_k,
        search_depth=search_depth,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    visualizations = write_primeqa_hybrid_bm25_miss_analysis_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid BM25 miss analysis report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "config": report["config"],
        "split_summary": {
            split: {
                "evaluated_questions": split_report["evaluated_questions"],
                "hit_at_top_k": split_report["hit_at_top_k"],
                "miss_count": split_report["miss_count"],
                "miss_rate": split_report["miss_rate"],
                "top_reason_tags": sorted(
                    split_report["reason_tag_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:8],
                "top_routes": sorted(
                    split_report["route_miss_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:8],
            }
            for split, split_report in report["split_reports"].items()
        },
        "cross_split_summary": report["cross_split_summary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
