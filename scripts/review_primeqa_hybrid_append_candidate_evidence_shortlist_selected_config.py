from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review as stage133_review,
)
from ts_rag_agent.config import ProjectSettings

review_selected_config = (
    stage133_review.review_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config
)
write_visualizations = (
    stage133_review
    .write_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_visualizations
)

app = typer.Typer(
    help="Review Stage133 PrimeQA hybrid append-candidate selected sidecar config."
)


@app.command()
def main(
    stage132_report: Annotated[
        Path | None,
        typer.Option("--stage132-report", help="Stage132 validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage133 selected-config review JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_review: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-review/--no-user-confirmed-review",
            help="Required confirmation for Stage133 selected-config review.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage133 selected sidecar config review report."""

    settings = ProjectSettings()
    stage132_report_path = stage132_report or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132.json"
    )
    output_path = output or (
        settings.artifact_dir
        / (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "selected_config_review_stage133.json"
        )
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "selected_config_review_stage133_visuals"
        )
    )

    report = review_selected_config(
        stage132_report_path=stage132_report_path,
        user_confirmed_review=user_confirmed_review,
        confirmation_note=confirmation_note,
    )
    visualizations = write_visualizations(
        report=report,
        output_dir=visualization_output_dir,
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
    typer.echo(f"Saved PrimeQA hybrid Stage133 selected-config review: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "review_id": report["review_id"],
        "stage132_summary": report["stage132_summary"],
        "selected_config_review": report["selected_config_review"],
        "replacement_route_review": report["replacement_route_review"],
        "agent_design_review": report["agent_design_review"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
