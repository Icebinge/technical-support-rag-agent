from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison import (
    run_primeqa_hybrid_evidence_answerability_comparison,
    write_primeqa_hybrid_evidence_answerability_comparison_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage105 PrimeQA hybrid evidence-answerability train/dev comparison."
)


@app.command()
def main(
    stage104_protocol: Annotated[
        Path | None,
        typer.Option("--stage104-protocol", help="Stage104 frozen protocol JSON."),
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
        typer.Option("--documents", help="PrimeQA corpus sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage105 comparison JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_comparison: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-comparison/--no-user-confirmed-comparison",
            help="Required confirmation for the Stage105 train/dev comparison.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    max_gold_window_sentences: Annotated[
        int,
        typer.Option(
            "--max-gold-window-sentences",
            help="Maximum contiguous gold-document sentence window for oracle span F1.",
        ),
    ] = 3,
    gold_span_gap_margin: Annotated[
        float,
        typer.Option(
            "--gold-span-gap-margin",
            help="F1 gap margin for gold-span-beats-selected-answer classification.",
        ),
    ] = 0.05,
    low_answer_f1_threshold: Annotated[
        float,
        typer.Option(
            "--low-answer-f1-threshold",
            help="Low-overlap answer F1 threshold for pipeline bucket classification.",
        ),
    ] = 0.2,
    sample_limit_per_bucket_transition: Annotated[
        int,
        typer.Option(
            "--sample-limit-per-bucket-transition",
            help="Public-safe changed-case samples per baseline-to-candidate bucket transition.",
        ),
    ] = 5,
) -> None:
    """Write the Stage105 train-selected, dev-validated comparison report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage104_protocol_path = stage104_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.json"
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
        / "primeqa_hybrid_evidence_answerability_comparison_stage105.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_stage105_visuals"
    )
    report = run_primeqa_hybrid_evidence_answerability_comparison(
        stage104_protocol_path=stage104_protocol_path,
        stage102_report_path=stage102_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_comparison=user_confirmed_comparison,
        confirmation_note=confirmation_note,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        sample_limit_per_bucket_transition=sample_limit_per_bucket_transition,
    )
    visualizations = write_primeqa_hybrid_evidence_answerability_comparison_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage105 comparison: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "user_confirmation": report["user_confirmation"],
        "split_contract": report["split_contract"],
        "analysis_config": report["analysis_config"],
        "data_summary": report["data_summary"],
        "baseline_result": {
            "config_id": report["baseline_result"]["config_id"],
            "weighted_target_scores_by_split": report["baseline_result"][
                "weighted_target_scores_by_split"
            ],
            "metrics_by_split": report["baseline_result"]["metrics_by_split"],
        },
        "train_selection": report["train_selection"],
        "dev_validation": report["dev_validation"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
