from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_second_wave_route_exhaustion_summary import (
    summarize_primeqa_hybrid_second_wave_route_exhaustion,
    write_primeqa_hybrid_second_wave_route_exhaustion_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage100 PrimeQA hybrid second-wave route exhaustion summary."
)


@app.command()
def main(
    stage83_report: Annotated[
        Path | None,
        typer.Option("--stage83-report", help="Stage83 first-wave exhaustion JSON."),
    ] = None,
    stage84_report: Annotated[
        Path | None,
        typer.Option("--stage84-report", help="Stage84 second-wave design JSON."),
    ] = None,
    stage87_report: Annotated[
        Path | None,
        typer.Option("--stage87-report", help="Stage87 LCDR stop JSON."),
    ] = None,
    stage90_report: Annotated[
        Path | None,
        typer.Option("--stage90-report", help="Stage90 structured query stop JSON."),
    ] = None,
    stage93_report: Annotated[
        Path | None,
        typer.Option("--stage93-report", help="Stage93 section signal stop JSON."),
    ] = None,
    stage96_report: Annotated[
        Path | None,
        typer.Option("--stage96-report", help="Stage96 score-margin stop JSON."),
    ] = None,
    stage99_report: Annotated[
        Path | None,
        typer.Option("--stage99-report", help="Stage99 selective dense+sparse stop JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage100 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_summary: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-summary/--no-user-confirmed-summary",
            help="Required confirmation for Stage100 route-exhaustion summary.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage100 summary from public-safe route outcome reports."""

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_route_exhaustion_summary_stage100.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals"
    )
    report = summarize_primeqa_hybrid_second_wave_route_exhaustion(
        stage83_report_path=stage83_report
        or settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json",
        stage84_report_path=stage84_report
        or settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json",
        stage87_report_path=stage87_report
        or settings.artifact_dir
        / "primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json",
        stage90_report_path=stage90_report
        or settings.artifact_dir
        / "primeqa_hybrid_structured_query_stop_decision_stage90.json",
        stage93_report_path=stage93_report
        or settings.artifact_dir
        / "primeqa_hybrid_section_signal_stop_decision_stage93.json",
        stage96_report_path=stage96_report
        or settings.artifact_dir
        / "primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json",
        stage99_report_path=stage99_report
        or settings.artifact_dir
        / "primeqa_hybrid_selective_dense_sparse_stop_decision_stage99.json",
        user_confirmed_summary=user_confirmed_summary,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_second_wave_route_exhaustion_visualizations(
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
    typer.echo(f"Saved PrimeQA second-wave route exhaustion summary: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "route_outcomes": report["route_outcomes"],
        "blocked_diagnostic": report["blocked_diagnostic"],
        "aggregate_summary": report["aggregate_summary"],
        "next_direction_options": report["next_direction_options"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
