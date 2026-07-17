from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    freeze_primeqa_hybrid_agent_tool_orchestration_protocol,
    write_primeqa_hybrid_agent_tool_orchestration_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage153 local Agent tool-orchestration protocol.")


@app.command()
def main(
    stage152_validation: Annotated[
        Path | None,
        typer.Option("--stage152-validation", help="Stage152 public validation JSON."),
    ] = None,
    stage139_validation: Annotated[
        Path | None,
        typer.Option("--stage139-validation", help="Stage139 public validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage153 public protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage153 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation before freezing the formal protocol.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage153 protocol report and ten visualizations."""

    artifact_dir = ProjectSettings().artifact_dir
    report = freeze_primeqa_hybrid_agent_tool_orchestration_protocol(
        stage152_validation_path=stage152_validation
        or artifact_dir / "primeqa_hybrid_agent_service_entrypoint_validation_stage152.json",
        stage139_validation_path=stage139_validation
        or artifact_dir
        / "primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_stage139.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_tool_orchestration_protocol_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage153 tool protocol: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "stage152_summary": report.get("stage152_summary"),
        "stage139_summary": report.get("stage139_summary"),
        "official_framework_research": report.get("official_framework_research"),
        "frozen_protocol": report.get("frozen_protocol"),
        "canonical_policy_evaluations": report.get("canonical_policy_evaluations"),
        "canonical_workflow_traces": report.get("canonical_workflow_traces"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
