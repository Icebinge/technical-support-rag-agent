from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    freeze_primeqa_hybrid_nondefault_runtime_activation_protocol,
    write_primeqa_hybrid_nondefault_runtime_activation_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze Stage141 strict non-default runtime activation protocol.")


@app.command()
def main(
    stage140_validation: Annotated[
        Path | None,
        typer.Option("--stage140-validation", help="Stage140 public-safe validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage141 public-safe protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage141 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the selected strict-C SLO.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
    selected_slo_profile_id: Annotated[
        str,
        typer.Option("--selected-slo-profile-id", help="Confirmed latency SLO profile id."),
    ] = "strict_c_warm_single_request_v1",
) -> None:
    """Write the aggregate-only Stage141 protocol and visualizations."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    report = freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
        stage140_validation_path=stage140_validation
        or artifact_dir
        / "primeqa_hybrid_online_candidate_pool_performance_validation_stage140.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
        selected_slo_profile_id=selected_slo_profile_id,
    )
    visualizations = write_primeqa_hybrid_nondefault_runtime_activation_protocol_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_nondefault_runtime_activation_protocol_stage141_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_nondefault_runtime_activation_protocol_stage141.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage141 activation protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "stage140_summary": report["stage140_summary"],
        "canonical_activation_evaluations": report["canonical_activation_evaluations"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
