from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_protocol import (
    freeze_primeqa_hybrid_agent_service_entrypoint_protocol,
    write_primeqa_hybrid_agent_service_entrypoint_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage151 local Agent service entrypoint protocol.")


@app.command()
def main(
    stage150_validation: Annotated[
        Path | None,
        typer.Option("--stage150-validation", help="Stage150 public-safe validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage151 public-safe protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage151 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage151 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the aggregate-only Stage151 protocol and visualizations."""

    artifact_dir = ProjectSettings().artifact_dir
    report = freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
        stage150_validation_path=stage150_validation
        or artifact_dir / "primeqa_hybrid_agent_http_transport_validation_stage150.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_service_entrypoint_protocol_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_service_entrypoint_protocol_stage151_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_service_entrypoint_protocol_stage151.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage151 service entrypoint protocol: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "stage150_summary": report.get("stage150_summary"),
        "frozen_protocol": report.get("frozen_protocol"),
        "canonical_policy_evaluations": report.get("canonical_policy_evaluations"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
