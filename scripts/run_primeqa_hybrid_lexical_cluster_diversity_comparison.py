from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_lexical_cluster_diversity_comparison import (
    run_primeqa_hybrid_lexical_cluster_diversity_comparison,
    write_primeqa_hybrid_lexical_cluster_diversity_comparison_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage86 train/dev-only PrimeQA hybrid LCDR comparison."
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
    stage85_report: Annotated[
        Path | None,
        typer.Option("--stage85-report", help="Stage85 LCDR protocol report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage86 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the frozen Stage85 protocol.",
        ),
    ] = False,
    confirmed_protocol_id: Annotated[
        str,
        typer.Option("--confirmed-protocol-id", help="Confirmed Stage85 protocol id."),
    ] = "lexical_cluster_diversity_rerank_train_dev_v1",
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage86 lexical cluster diversity comparison report."""

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
    stage85_report_path = stage85_report or (
        settings.artifact_dir
        / "primeqa_hybrid_lexical_cluster_diversity_protocol_stage85.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_lexical_cluster_diversity_comparison_stage86.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals"
    )

    report = run_primeqa_hybrid_lexical_cluster_diversity_comparison(
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        stage75_report_path=stage75_report_path,
        stage85_report_path=stage85_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmed_protocol_id=confirmed_protocol_id,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_lexical_cluster_diversity_comparison_visualizations(
            report=report,
            output_dir=visualization_output_dir,
        )
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
    typer.echo(f"Saved PrimeQA hybrid LCDR comparison report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "config": report["config"],
        "candidate_configs": report["candidate_configs"],
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
