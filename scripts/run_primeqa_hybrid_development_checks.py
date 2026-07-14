from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_development_checks import (
    run_primeqa_hybrid_development_checks,
    write_primeqa_hybrid_development_check_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage70 train/dev development checks on the frozen PrimeQA hybrid split."
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
        typer.Option("--candidate-dataset", help="Stage69 candidate JSONL path."),
    ] = None,
    candidate_summary: Annotated[
        Path | None,
        typer.Option("--candidate-summary", help="Stage69 candidate summary JSON path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage70 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="Comma-separated BM25 top-k values."),
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
    """Run Stage70 development-only checks and write the report."""

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
    candidate_summary_path = candidate_summary or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.summary.json"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_development_checks_stage70.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_development_checks_stage70_visuals"
    )

    for path in [
        train_path,
        dev_path,
        documents_path,
        candidate_dataset_path,
        candidate_summary_path,
    ]:
        _ensure_file_exists(path)

    report = run_primeqa_hybrid_development_checks(
        train_split_path=train_path,
        dev_split_path=dev_path,
        documents_path=documents_path,
        candidate_dataset_path=candidate_dataset_path,
        candidate_summary_path=candidate_summary_path,
        top_k_values=_parse_top_k_values(top_k),
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    visualizations = write_primeqa_hybrid_development_check_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid development checks report: {output_path}")


def _parse_top_k_values(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise typer.BadParameter("--top-k must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise typer.BadParameter("--top-k values must be positive integers.")
    return values


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "bm25_baseline": report["bm25_baseline"],
        "candidate_artifact_checks": report["candidate_artifact_checks"],
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


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
