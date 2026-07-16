from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_append_candidate_evidence_shortlist_protocol import (
    freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol,
    write_primeqa_hybrid_append_candidate_evidence_shortlist_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage131 PrimeQA hybrid append-candidate evidence shortlist protocol."
)


@app.command()
def main(
    stage130_review: Annotated[
        Path | None,
        typer.Option("--stage130-review", help="Stage130 failure-review JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage131 protocol JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for Stage131 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage131 append-candidate evidence shortlist protocol report."""

    settings = ProjectSettings()
    stage130_review_path = stage130_review or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_integration_failure_review_stage130.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals"
    )

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_review_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_append_candidate_evidence_shortlist_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage131 protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "stage130_summary": report["stage130_summary"],
        "frozen_protocol": report["frozen_protocol"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
