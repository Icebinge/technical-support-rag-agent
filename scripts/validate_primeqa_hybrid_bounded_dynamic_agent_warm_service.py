from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_warm_service_validation import (
    validate_primeqa_hybrid_bounded_dynamic_agent_warm_service,
    write_stage159_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate Stage159 full-dev warm bounded Agent service behavior.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", min=1024, max=65535, help="Available loopback port."),
    ] = 18159,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage159 public validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage159 SVG output directory."),
    ] = None,
) -> None:
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    report = validate_primeqa_hybrid_bounded_dynamic_agent_warm_service(
        settings=settings,
        port=port,
        user_confirmed_full_dev_protocol=True,
        progress_sink=_write_progress,
    )
    visualizations = write_stage159_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_warm_service_stage159_visuals"
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
        artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_warm_service_stage159.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage159 warm service validation: {output_path}")
    if report.get("decision", {}).get("all_guards_passed") is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    real_service = report.get("real_service") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "dev_query_protocol": report.get("dev_query_protocol"),
        "workload_plan": report.get("workload_plan"),
        "startup": report.get("startup"),
        "dev_turn_summary": real_service.get("dev_turn_summary"),
        "capacity_probe": real_service.get("capacity_probe"),
        "coordinator_counters_after_shutdown": real_service.get(
            "coordinator_counters_after_shutdown"
        ),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "public_safe_contract": report.get("public_safe_contract"),
        "decision": report.get("decision"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
