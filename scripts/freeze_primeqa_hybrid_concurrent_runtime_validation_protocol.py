from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    freeze_primeqa_hybrid_concurrent_runtime_validation_protocol,
    write_primeqa_hybrid_concurrent_runtime_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze Stage144 strict practical B concurrency validation protocol.")


@app.command()
def main(
    stage143_validation: Annotated[
        Path | None,
        typer.Option("--stage143-validation", help="Stage143 public-safe validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage144 public-safe protocol JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage144 SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for strict practical profile B.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
    selected_profile_id: Annotated[
        str,
        typer.Option("--selected-profile-id", help="Confirmed concurrency profile id."),
    ] = "strict_practical_b_concurrency4_v1",
) -> None:
    """Write the aggregate-only Stage144 protocol and visualizations."""

    artifact_dir = ProjectSettings().artifact_dir
    report = freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
        stage143_validation_path=stage143_validation
        or artifact_dir / "primeqa_hybrid_optional_sidecar_runtime_validation_stage143.json",
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
        selected_profile_id=selected_profile_id,
    )
    visualizations = write_primeqa_hybrid_concurrent_runtime_protocol_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_concurrent_runtime_protocol_stage144_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_concurrent_runtime_validation_protocol_stage144.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage144 concurrency protocol: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = report.get("frozen_protocol") or {}
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "benchmark_machine": report["benchmark_machine"],
        "profile": protocol.get("profile"),
        "train_validation_contract": protocol.get("train_validation_contract"),
        "overload_contract": protocol.get("overload_contract"),
        "dev_report_only_contract": protocol.get("dev_report_only_contract"),
        "canonical_validation_evaluations": report["canonical_validation_evaluations"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
