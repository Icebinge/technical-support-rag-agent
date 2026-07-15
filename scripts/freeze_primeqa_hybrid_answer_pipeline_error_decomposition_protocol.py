from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_protocol import (
    freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol,
    write_primeqa_hybrid_answer_pipeline_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage101 PrimeQA hybrid answer-pipeline decomposition protocol."
)


@app.command()
def main(
    stage100_report: Annotated[
        Path | None,
        typer.Option("--stage100-report", help="Stage100 route exhaustion JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage101 protocol report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage101 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage101 answer-pipeline error decomposition protocol report."""

    settings = ProjectSettings()
    stage100_report_path = stage100_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_route_exhaustion_summary_stage100.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals"
    )
    report = freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
        stage100_report_path=stage100_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_answer_pipeline_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid answer-pipeline protocol report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stage100_summary": report["stage100_summary"],
        "protocol_id": frozen["protocol_id"],
        "bucket_assignment_contract": frozen["bucket_assignment_contract"],
        "public_safe_output_contract": frozen["public_safe_output_contract"],
        "train_dev_execution_rule": frozen["train_dev_execution_rule"],
        "fallback_strategy_policy": frozen["fallback_strategy_policy"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
