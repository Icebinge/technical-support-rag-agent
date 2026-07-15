from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_score_margin_bm25_stop_decision import (
    decide_primeqa_hybrid_score_margin_bm25_stop,
    write_primeqa_hybrid_score_margin_bm25_stop_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Write Stage96 PrimeQA hybrid score-margin BM25 stop decision.")


@app.command()
def main(
    stage84_report: Annotated[
        Path | None,
        typer.Option("--stage84-report", help="Stage84 second-wave design JSON."),
    ] = None,
    stage95_report: Annotated[
        Path | None,
        typer.Option("--stage95-report", help="Stage95 score-margin BM25 JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage96 stop decision JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_stop: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stop/--no-user-confirmed-stop",
            help="Required confirmation for stopping the score-margin BM25 route.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage96 score-margin BM25 stop decision from public-safe reports."""

    settings = ProjectSettings()
    stage84_report_path = stage84_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json"
    )
    stage95_report_path = stage95_report or (
        settings.artifact_dir
        / "primeqa_hybrid_score_margin_bm25_comparison_stage95.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals"
    )

    report = decide_primeqa_hybrid_score_margin_bm25_stop(
        stage84_report_path=stage84_report_path,
        stage95_report_path=stage95_report_path,
        user_confirmed_stop=user_confirmed_stop,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_score_margin_bm25_stop_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid score-margin BM25 stop decision: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stopped_route": report["stopped_route"],
        "candidate_queue": report["candidate_queue"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
