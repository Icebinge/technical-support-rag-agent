from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade_validation import (
    run_primeqa_hybrid_agent_request_facade_validation,
    write_primeqa_hybrid_agent_request_facade_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate the Stage148 transport-neutral Agent request facade.")


@app.command()
def main(
    stage147_protocol: Annotated[
        Path | None,
        typer.Option("--stage147-protocol", help="Stage147 public-safe protocol JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage148 public-safe validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage148 SVG charts."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required confirmation for Stage148 synthetic validation.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the aggregate/synthetic-only Stage148 report and visualizations."""

    artifact_dir = ProjectSettings().artifact_dir
    report = run_primeqa_hybrid_agent_request_facade_validation(
        stage147_protocol_path=stage147_protocol
        or artifact_dir / "primeqa_hybrid_agent_request_facade_protocol_stage147.json",
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_request_facade_validation_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_request_facade_validation_stage148_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_request_facade_validation_stage148.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage148 facade validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "source_gate_checks": report.get("source_gate_checks"),
        "source_gate_passed": report.get("source_gate_passed"),
        "synthetic_validation_executed": report.get("synthetic_validation_executed"),
        "synthetic_validation": report.get("synthetic_validation"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
