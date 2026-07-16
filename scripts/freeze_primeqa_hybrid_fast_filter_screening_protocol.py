from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_protocol import (
    freeze_primeqa_hybrid_fast_filter_screening_protocol,
    write_primeqa_hybrid_fast_filter_screening_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage120 PrimeQA hybrid fast-filter screening protocol."
)


@app.command()
def main(
    stage119_report: Annotated[
        Path | None,
        typer.Option("--stage119-report", help="Stage119 stop decision JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage120 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for freezing this fast-filter protocol.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage120 fast-filter screening protocol freeze report."""

    settings = ProjectSettings()
    stage119_report_path = stage119_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_stage_reranking_stop_decision_stage119.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_protocol_stage120.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals"
    )

    report = freeze_primeqa_hybrid_fast_filter_screening_protocol(
        stage119_report_path=stage119_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_fast_filter_screening_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage120 protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "stage119_summary": report["stage119_summary"],
        "fixed_candidate_pool_contract": frozen["fixed_candidate_pool_contract"],
        "fast_filter_contract": frozen["fast_filter_contract"],
        "candidate_families": frozen["candidate_families"],
        "candidate_configs": frozen["candidate_configs"],
        "selection_rules": frozen["selection_rules"],
        "feature_contract": frozen["feature_contract"],
        "blocked_options": frozen["blocked_options"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
