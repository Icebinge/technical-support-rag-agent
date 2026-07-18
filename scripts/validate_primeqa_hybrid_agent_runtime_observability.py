from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability_validation import (
    run_primeqa_hybrid_agent_runtime_observability_validation,
    write_primeqa_hybrid_agent_runtime_observability_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate Stage155 Agent activation and observability.")


@app.command()
def main(
    port: Annotated[
        int,
        typer.Option("--port", min=1024, max=65535, help="Exact loopback port."),
    ],
    stage154_validation: Annotated[
        Path | None,
        typer.Option("--stage154-validation", help="Stage154 formal validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage155 public-safe validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage155 SVGs."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required before the one real resource and HTTP lifecycle.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage155 report and ten SVG charts."""

    root = Path(__file__).resolve().parents[1]
    base_settings = ProjectSettings()
    settings = ProjectSettings(
        data_dir=base_settings.data_dir,
        artifact_dir=base_settings.artifact_dir,
        enable_optional_sidecar_agent=False,
        enable_concurrent_sidecar_agent=True,
        enable_local_agent_http_transport=True,
    )
    artifact_dir = settings.artifact_dir
    application_dir = root / "src" / "ts_rag_agent" / "application"
    report = run_primeqa_hybrid_agent_runtime_observability_validation(
        stage154_validation_path=stage154_validation
        or artifact_dir / "primeqa_hybrid_agent_tool_workflow_validation_stage154.json",
        stage153_protocol_path=(
            artifact_dir / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
        ),
        pyproject_path=root / "pyproject.toml",
        workflow_source_path=application_dir / "primeqa_hybrid_agent_tool_workflow.py",
        concurrent_runtime_source_path=(
            application_dir / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
        ),
        observability_source_path=(
            application_dir / "primeqa_hybrid_agent_runtime_observability.py"
        ),
        service_entrypoint_source_path=(
            application_dir / "primeqa_hybrid_agent_service_entrypoint.py"
        ),
        settings=settings,
        port=port,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_runtime_observability_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_runtime_observability_stage155_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_runtime_observability_stage155.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage155 Agent observability validation: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "stage154_summary": report.get("stage154_summary"),
        "activation_validation": report.get("activation_validation"),
        "synthetic_validation": report.get("synthetic_validation"),
        "real_resource_service_lifecycle": report.get("real_resource_service_lifecycle"),
        "real_observation": report.get("real_observation"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "timing_seconds": report.get("timing_seconds"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
