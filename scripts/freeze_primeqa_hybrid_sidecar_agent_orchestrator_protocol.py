from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_protocol import (
    freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol,
    write_primeqa_hybrid_sidecar_agent_orchestrator_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage136 sidecar agent orchestrator and Stage137 validation protocol."
)


@app.command()
def main(
    stage135_validation: Annotated[
        Path | None,
        typer.Option("--stage135-validation", help="Stage135 validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage136 protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage136 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the public-safe Stage136 orchestrator protocol report."""

    settings = ProjectSettings()
    stage135_validation_path = stage135_validation or (
        settings.artifact_dir
        / "primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_"
        "validation_stage135.json"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals"
    )

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=stage135_validation_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_sidecar_agent_orchestrator_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage136 orchestrator protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "stage135_summary": report["stage135_summary"],
        "frozen_protocol": report["frozen_protocol"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
