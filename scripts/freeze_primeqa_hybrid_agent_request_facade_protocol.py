from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade_protocol import (
    freeze_primeqa_hybrid_agent_request_facade_protocol,
    write_primeqa_hybrid_agent_request_facade_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage147 transport-neutral Agent request-facade protocol.")


@app.command()
def main(
    stage146_validation: Annotated[
        Path | None,
        typer.Option("--stage146-validation", help="Stage146 public-safe validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage147 public-safe protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage147 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage147 facade contract.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the aggregate/specification-only Stage147 report and SVGs."""

    artifact_dir = ProjectSettings().artifact_dir
    report = freeze_primeqa_hybrid_agent_request_facade_protocol(
        stage146_validation_path=stage146_validation
        or artifact_dir / "primeqa_hybrid_concurrent_runtime_activation_validation_stage146.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_request_facade_protocol_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_request_facade_protocol_stage147_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_request_facade_protocol_stage147.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage147 facade protocol: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = report.get("frozen_protocol") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "user_confirmation": report.get("user_confirmation"),
        "stage146_summary": report.get("stage146_summary"),
        "private_call_contract": protocol.get("private_call_contract"),
        "public_telemetry_contract": protocol.get("public_telemetry_contract"),
        "error_contract": protocol.get("error_contract"),
        "cancellation_contract": protocol.get("cancellation_contract"),
        "lifecycle_contract": protocol.get("lifecycle_contract"),
        "shutdown_contract": protocol.get("shutdown_contract"),
        "canonical_policy_evaluations": report.get("canonical_policy_evaluations"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
