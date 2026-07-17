from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_validation import (
    run_primeqa_hybrid_agent_service_entrypoint_validation,
    write_primeqa_hybrid_agent_service_entrypoint_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate the Stage152 local Agent service entrypoint.")


@app.command()
def main(
    port: Annotated[
        int,
        typer.Option("--port", min=1024, max=65535, help="Exact Stage152 loopback port."),
    ],
    stage151_protocol: Annotated[
        Path | None,
        typer.Option("--stage151-protocol", help="Stage151 public-safe protocol JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage152 public-safe validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage152 SVG charts."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required before the one real local resource/service lifecycle.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage152 implementation validation report and visualizations."""

    base_settings = ProjectSettings()
    settings = ProjectSettings(
        data_dir=base_settings.data_dir,
        artifact_dir=base_settings.artifact_dir,
        enable_optional_sidecar_agent=False,
        enable_concurrent_sidecar_agent=True,
        enable_local_agent_http_transport=True,
    )
    artifact_dir = settings.artifact_dir
    report = run_primeqa_hybrid_agent_service_entrypoint_validation(
        stage151_protocol_path=stage151_protocol
        or artifact_dir / "primeqa_hybrid_agent_service_entrypoint_protocol_stage151.json",
        settings=settings,
        port=port,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_service_entrypoint_validation_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_service_entrypoint_validation_stage152_visuals",
    )
    report = {
        **report,
        "visualizations": [asdict_visualization(row) for row in visualizations],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_service_entrypoint_validation_stage152.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage152 service validation: {output_path}")


def asdict_visualization(visualization: Any) -> dict[str, str]:
    return {"name": visualization.name, "path": visualization.path}


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "source_gate_passed": report.get("source_gate_passed"),
        "synthetic_composition_cases": report.get("synthetic_composition_cases"),
        "real_resource_service_lifecycle": report.get("real_resource_service_lifecycle"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
