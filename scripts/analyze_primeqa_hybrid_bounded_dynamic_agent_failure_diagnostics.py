from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_validation as stage160,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage160 dev-only bounded Agent failure diagnostics.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", min=1024, max=65535, help="Available loopback port."),
    ] = 18160,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage160 public aggregate JSON."),
    ] = None,
    private_output: Annotated[
        Path | None,
        typer.Option("--private-output", help="Ignored hashed case diagnostic JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage160 SVG output directory."),
    ] = None,
) -> None:
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    result = stage160.validate_primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics(
        settings=settings,
        port=port,
        user_confirmed_dev_gold_diagnostics=True,
        progress_sink=_write_progress,
    )
    private_path = private_output or (
        artifact_dir
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160_private.json"
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(result.private_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    report = result.public_report
    visualizations = stage160.write_stage160_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifact_dir
            / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160_visuals"
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
        artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage160 public diagnostics: {output_path}")
    typer.echo(f"Saved Stage160 private hashed diagnostics: {private_path}")
    if report.get("decision", {}).get("all_guards_passed") is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    service = report.get("real_service") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "dev_diagnostic_protocol": report.get("dev_diagnostic_protocol"),
        "workload_plan": report.get("workload_plan"),
        "grouped_fold_protocol": report.get("grouped_fold_protocol"),
        "startup": report.get("startup"),
        "dev_http": service.get("dev_http"),
        "aggregate_diagnostics": report.get("aggregate_diagnostics"),
        "private_diagnostic_artifact_contract": report.get("private_diagnostic_artifact_contract"),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "public_safe_contract": report.get("public_safe_contract"),
        "decision": report.get("decision"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
