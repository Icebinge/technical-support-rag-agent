from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_retrieval_context_miss_audit_protocol import (
    freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol,
    write_primeqa_hybrid_retrieval_context_miss_audit_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage111 PrimeQA hybrid retrieval-context-miss audit protocol."
)


@app.command()
def main(
    stage102_report: Annotated[
        Path | None,
        typer.Option("--stage102-report", help="Stage102 decomposition report JSON."),
    ] = None,
    stage107_report: Annotated[
        Path | None,
        typer.Option("--stage107-report", help="Stage107 failure-pattern report JSON."),
    ] = None,
    stage110_report: Annotated[
        Path | None,
        typer.Option("--stage110-report", help="Stage110 stop decision JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage111 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for freezing this audit protocol.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage111 retrieval-context-miss audit protocol freeze report."""

    settings = ProjectSettings()
    stage102_report_path = stage102_report or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json"
    )
    stage107_report_path = stage107_report or (
        settings.artifact_dir
        / "primeqa_hybrid_validation_failure_pattern_analysis_stage107.json"
    )
    stage110_report_path = stage110_report or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals"
    )

    report = freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol(
        stage102_report_path=stage102_report_path,
        stage107_report_path=stage107_report_path,
        stage110_report_path=stage110_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_retrieval_context_miss_audit_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage111 audit protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "route_id": report["route_id"],
        "user_confirmation": report["user_confirmation"],
        "stage102_summary": report["stage102_summary"],
        "stage107_summary": report["stage107_summary"],
        "stage110_summary": report["stage110_summary"],
        "audit_dimensions": frozen["audit_dimensions"],
        "stage112_run_contract": frozen["stage112_run_contract"],
        "public_safe_output_contract": frozen["public_safe_output_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
