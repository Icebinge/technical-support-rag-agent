from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_retrieval_index_redesign_comparison import (
    run_primeqa_hybrid_retrieval_index_redesign_comparison,
    write_primeqa_hybrid_retrieval_index_redesign_comparison_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage114 PrimeQA hybrid retrieval/index redesign train-CV/dev comparison."
)


@app.command()
def main(
    stage113_protocol: Annotated[
        Path | None,
        typer.Option("--stage113-protocol", help="Stage113 frozen protocol JSON."),
    ] = None,
    stage102_report: Annotated[
        Path | None,
        typer.Option("--stage102-report", help="Stage102 decomposition report JSON."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Stage68 train split JSONL."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Stage68 dev split JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training/dev sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage114 comparison JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_comparison: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-comparison/--no-user-confirmed-comparison",
            help="Required confirmation for the Stage114 train-CV/dev comparison.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="Retrieval depth for each run."),
    ] = 10,
    component_depth: Annotated[
        int,
        typer.Option(
            "--component-depth",
            help="Internal retrieval depth for RRF and special-token candidate components.",
        ),
    ] = 50,
) -> None:
    """Write the Stage114 train-CV-selected, dev-reported comparison."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage113_protocol_path = stage113_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_index_redesign_protocol_stage113.json"
    )
    stage102_report_path = stage102_report or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json"
    )
    train_split_path = train_split or (
        split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    )
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_index_redesign_comparison_stage114.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals"
    )

    report = run_primeqa_hybrid_retrieval_index_redesign_comparison(
        stage113_protocol_path=stage113_protocol_path,
        stage102_report_path=stage102_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_comparison=user_confirmed_comparison,
        confirmation_note=confirmation_note,
        retrieval_top_k=retrieval_top_k,
        component_depth=component_depth,
    )
    visualizations = write_primeqa_hybrid_retrieval_index_redesign_comparison_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage114 comparison: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "split_contract": report["split_contract"],
        "analysis_config": report["analysis_config"],
        "data_summary": report["data_summary"],
        "baseline_result": {
            "config_id": report["baseline_result"]["config_id"],
            "retrieval_metrics_by_split": report["baseline_result"][
                "retrieval_metrics_by_split"
            ],
            "metrics_by_split": report["baseline_result"]["metrics_by_split"],
        },
        "train_cv_selection": report["train_cv_selection"],
        "dev_validation": report["dev_validation"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
