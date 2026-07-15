from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_protocol import (
    freeze_primeqa_hybrid_failure_pattern_redesign_protocol,
    write_primeqa_hybrid_failure_pattern_redesign_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage108 PrimeQA hybrid failure-pattern redesign protocol."
)


@app.command()
def main(
    stage107_report: Annotated[
        Path | None,
        typer.Option("--stage107-report", help="Stage107 failure-pattern JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage108 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for Stage108 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage108 failure-pattern redesign protocol freeze report."""

    settings = ProjectSettings()
    stage107_report_path = stage107_report or (
        settings.artifact_dir
        / "primeqa_hybrid_validation_failure_pattern_analysis_stage107.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_protocol_stage108.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals"
    )

    report = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=stage107_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_failure_pattern_redesign_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage108 redesign protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "stage107_summary": report["stage107_summary"],
        "candidate_families": frozen["candidate_families"],
        "candidate_config_grid": frozen["candidate_config_grid"],
        "train_selection_rule": frozen["train_selection_rule"],
        "dev_validation_rule": frozen["dev_validation_rule"],
        "metric_contract": frozen["metric_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
