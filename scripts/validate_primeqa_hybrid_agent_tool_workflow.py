from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow_validation import (
    run_primeqa_hybrid_agent_tool_workflow_validation,
    write_primeqa_hybrid_agent_tool_workflow_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate the Stage154 LangGraph Agent tool workflow.")


@app.command()
def main(
    stage153_protocol: Annotated[
        Path | None,
        typer.Option("--stage153-protocol", help="Stage153 public protocol JSON."),
    ] = None,
    stage152_support_validation: Annotated[
        Path | None,
        typer.Option(
            "--stage152-support-validation",
            help="Current-code Stage152 real lifecycle support JSON.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage154 public validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage154 SVG charts."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required before accepting the formal Stage154 validation.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage154 implementation validation and ten SVG charts."""

    root = Path(__file__).resolve().parents[1]
    artifact_dir = ProjectSettings().artifact_dir
    report = run_primeqa_hybrid_agent_tool_workflow_validation(
        stage153_protocol_path=stage153_protocol
        or artifact_dir / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json",
        pyproject_path=root / "pyproject.toml",
        workflow_source_path=(
            root / "src" / "ts_rag_agent" / "application" / "primeqa_hybrid_agent_tool_workflow.py"
        ),
        concurrent_runtime_source_path=(
            root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
        ),
        stage152_support_validation_path=stage152_support_validation,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_tool_workflow_validation_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_agent_tool_workflow_validation_stage154_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_agent_tool_workflow_validation_stage154.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage154 workflow validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "stage153_summary": report.get("stage153_summary"),
        "dependency_evidence": report.get("dependency_evidence"),
        "implementation_contract": report.get("implementation_contract"),
        "synthetic_validation": report.get("synthetic_validation"),
        "stage152_current_service_support": report.get("stage152_current_service_support"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
