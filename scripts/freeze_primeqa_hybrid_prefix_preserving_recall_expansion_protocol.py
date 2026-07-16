from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_prefix_preserving_recall_expansion_protocol import (
    freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol,
    write_primeqa_hybrid_prefix_preserving_recall_expansion_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage125 PrimeQA hybrid prefix-preserving recall expansion protocol."
)


@app.command()
def main(
    stage124_report: Annotated[
        Path | None,
        typer.Option("--stage124-report", help="Stage124 validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage125 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for freezing this prefix-preserving protocol.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage125 prefix-preserving recall expansion protocol report."""

    settings = ProjectSettings()
    stage124_report_path = stage124_report or (
        settings.artifact_dir
        / "primeqa_hybrid_first_stage_recall_expansion_validation_stage124.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals"
    )

    report = freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol(
        stage124_report_path=stage124_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_prefix_preserving_recall_expansion_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage125 protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "stage124_summary": report["stage124_summary"],
        "baseline_prefix_contract": frozen["baseline_prefix_contract"],
        "append_generation_contract": frozen["append_generation_contract"],
        "candidate_families": frozen["candidate_families"],
        "candidate_configs": frozen["candidate_configs"],
        "selection_rules": frozen["selection_rules"],
        "blocked_options": frozen["blocked_options"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()

