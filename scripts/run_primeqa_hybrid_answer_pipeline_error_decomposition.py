from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    run_primeqa_hybrid_answer_pipeline_error_decomposition,
    write_primeqa_hybrid_answer_pipeline_decomposition_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage102 PrimeQA hybrid answer-pipeline error decomposition."
)


@app.command()
def main(
    stage101_protocol: Annotated[
        Path | None,
        typer.Option("--stage101-protocol", help="Stage101 protocol JSON."),
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
        typer.Option("--documents", help="PrimeQA corpus sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage102 analysis JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_analysis: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-analysis/--no-user-confirmed-analysis",
            help="Required confirmation for the Stage102 train/dev analysis.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="BM25 context depth for the analysis."),
    ] = 10,
    min_evidence_score: Annotated[
        float,
        typer.Option("--min-evidence-score", help="Verifier evidence threshold."),
    ] = 7.0,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="Sanitized samples per bucket."),
    ] = 5,
) -> None:
    """Write Stage102 public-safe train/dev answer-pipeline decomposition."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage101_protocol_path = stage101_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.json"
    )
    train_split_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals"
    )
    report = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=stage101_protocol_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_analysis=user_confirmed_analysis,
        confirmation_note=confirmation_note,
        retrieval_top_k=retrieval_top_k,
        min_evidence_score=min_evidence_score,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    visualizations = write_primeqa_hybrid_answer_pipeline_decomposition_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid answer-pipeline decomposition: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "user_confirmation": report["user_confirmation"],
        "analysis_config": report["analysis_config"],
        "data_summary": report["data_summary"],
        "metrics_by_split": report["metrics_by_split"],
        "aggregate_outputs": {
            "bucket_counts_by_split": report["aggregate_outputs"][
                "bucket_counts_by_split"
            ],
            "top_priority_buckets": report["aggregate_outputs"][
                "top_priority_buckets"
            ],
            "verification_decision_distributions": report["aggregate_outputs"][
                "verification_decision_distributions"
            ],
        },
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
