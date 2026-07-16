from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_protocol import (
    freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol,
    write_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help=(
        "Freeze Stage134 PrimeQA hybrid Stage116 answer-context plus "
        "Stage128 sidecar-observation agent protocol."
    )
)


@app.command()
def main(
    stage128_protocol: Annotated[
        Path | None,
        typer.Option("--stage128-protocol", help="Stage128 protocol JSON."),
    ] = None,
    stage129_validation: Annotated[
        Path | None,
        typer.Option("--stage129-validation", help="Stage129 validation JSON."),
    ] = None,
    stage133_review: Annotated[
        Path | None,
        typer.Option("--stage133-review", help="Stage133 selected sidecar review JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage134 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for Stage134 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage134 sidecar-observation agent protocol report."""

    settings = ProjectSettings()
    stage128_protocol_path = stage128_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
    )
    stage129_validation_path = stage129_validation or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_retrieval_integration_validation_stage129.json"
    )
    stage133_review_path = stage133_review or (
        settings.artifact_dir
        / (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "selected_config_review_stage133.json"
        )
    )
    output_path = output or (
        settings.artifact_dir
        / (
            "primeqa_hybrid_stage116_answer_context_stage128_sidecar_"
            "protocol_stage134.json"
        )
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / (
            "primeqa_hybrid_stage116_answer_context_stage128_sidecar_"
            "protocol_stage134_visuals"
        )
    )

    report = freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
        stage128_protocol_path=stage128_protocol_path,
        stage129_validation_path=stage129_validation_path,
        stage133_review_path=stage133_review_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_visualizations(
            report=report,
            output_dir=visualization_output_dir,
        )
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage134 sidecar protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "stage128_summary": report["stage128_summary"],
        "stage129_summary": report["stage129_summary"],
        "stage133_summary": report["stage133_summary"],
        "frozen_protocol": report["frozen_protocol"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
