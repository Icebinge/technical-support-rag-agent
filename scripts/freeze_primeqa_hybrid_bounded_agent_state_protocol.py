from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    freeze_primeqa_hybrid_bounded_agent_state_protocol,
    write_primeqa_hybrid_bounded_agent_state_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage156 bounded Agent decision/state protocol.")


@app.command()
def main(
    stage155_validation: Annotated[
        Path | None,
        typer.Option("--stage155-validation", help="Stage155 public validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage156 public protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage156 SVG charts."),
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
    artifact_dir = ProjectSettings().artifact_dir
    report = freeze_primeqa_hybrid_bounded_agent_state_protocol(
        stage155_validation_path=stage155_validation
        or artifact_dir / "primeqa_hybrid_agent_runtime_observability_stage155.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_bounded_agent_state_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_bounded_agent_state_protocol_stage156_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_bounded_agent_state_protocol_stage156.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage156 bounded Agent protocol: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "stage155_summary": report.get("stage155_summary"),
        "bounded_decision_contract": report.get("bounded_decision_contract"),
        "volatile_thread_state_contract": report.get("volatile_thread_state_contract"),
        "canonical_dynamic_policy_cases": report.get("canonical_dynamic_policy_cases"),
        "canonical_thread_state_cases": report.get("canonical_thread_state_cases"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
