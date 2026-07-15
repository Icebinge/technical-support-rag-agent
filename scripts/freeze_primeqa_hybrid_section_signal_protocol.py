from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_section_signal_protocol import (
    freeze_primeqa_hybrid_section_signal_protocol,
    write_primeqa_hybrid_section_signal_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze Stage91 PrimeQA hybrid section-signal protocol.")


@app.command()
def main(
    stage84_report: Annotated[
        Path | None,
        typer.Option("--stage84-report", help="Stage84 second-wave design JSON."),
    ] = None,
    stage90_report: Annotated[
        Path | None,
        typer.Option("--stage90-report", help="Stage90 stop decision JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage91 protocol report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_candidate: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-candidate/--no-user-confirmed-candidate",
            help="Required confirmation for the Stage90 next candidate.",
        ),
    ] = False,
    confirmed_candidate_id: Annotated[
        str,
        typer.Option("--confirmed-candidate-id", help="Confirmed Stage90 candidate id."),
    ] = "section_signal_guarded_expansion_design",
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage91 section-signal protocol freeze report."""

    settings = ProjectSettings()
    stage84_report_path = stage84_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json"
    )
    stage90_report_path = stage90_report or (
        settings.artifact_dir
        / "primeqa_hybrid_structured_query_stop_decision_stage90.json"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_section_signal_protocol_stage91.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_section_signal_protocol_stage91_visuals"
    )

    report = freeze_primeqa_hybrid_section_signal_protocol(
        stage84_report_path=stage84_report_path,
        stage90_report_path=stage90_report_path,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_section_signal_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid section-signal protocol report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "protocol_id": frozen["protocol_id"],
        "candidate_id": frozen["candidate_id"],
        "candidate_config_grid": frozen["candidate_config_grid"],
        "train_selection_rule": frozen["train_selection_rule"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
