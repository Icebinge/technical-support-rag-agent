from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_validation import (
    validate_primeqa_hybrid_bounded_dynamic_agent_service,
    write_stage158_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate the Stage158 bounded dynamic Agent local service.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", min=1024, max=65535, help="Available loopback port."),
    ] = 18158,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage158 public validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage158 SVG output directory."),
    ] = None,
) -> None:
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    report = validate_primeqa_hybrid_bounded_dynamic_agent_service(
        settings=settings,
        port=port,
        user_confirmed_protocol_a=True,
    )
    visualizations = write_stage158_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_service_stage158_visuals"
        ),
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_service_stage158.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage158 bounded dynamic Agent service validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "environment": report.get("environment"),
        "synthetic_service_cases": report.get("synthetic_service_cases"),
        "real_service": report.get("real_service"),
        "startup": report.get("startup"),
        "guard_checks": report.get("guard_checks"),
        "public_safe_contract": report.get("public_safe_contract"),
        "decision": report.get("decision"),
        "timing_seconds": report.get("timing_seconds"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
