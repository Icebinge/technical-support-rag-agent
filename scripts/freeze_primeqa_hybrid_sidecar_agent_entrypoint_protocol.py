from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_entrypoint_protocol import (
    freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol,
    write_primeqa_hybrid_sidecar_agent_entrypoint_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the optional Stage138 sidecar-agent entrypoint protocol.")


@app.command()
def main(
    stage137_validation: Annotated[
        Path | None,
        typer.Option("--stage137-validation", help="Stage137 validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage138 protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage138 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage138 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the public-safe Stage138 entrypoint protocol report."""

    settings = ProjectSettings()
    stage137_validation_path = stage137_validation or (
        settings.artifact_dir / "primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_stage138.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_stage138_visuals"
    )

    report = freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
        stage137_validation_path=stage137_validation_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_sidecar_agent_entrypoint_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage138 entrypoint protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "stage137_summary": report["stage137_summary"],
        "frozen_protocol": report["frozen_protocol"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
