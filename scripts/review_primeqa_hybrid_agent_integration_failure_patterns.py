from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_agent_integration_failure_review import (
    review_primeqa_hybrid_agent_integration_failure_patterns,
    write_primeqa_hybrid_agent_integration_failure_review_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Review Stage130 PrimeQA hybrid agent integration failures.")


@app.command()
def main(
    stage129_report: Annotated[
        Path | None,
        typer.Option("--stage129-report", help="Stage129 validation JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage130 review JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_review: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-review/--no-user-confirmed-review",
            help="Required confirmation for Stage130 failure-pattern review.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage130 public-safe failure-pattern review report."""

    settings = ProjectSettings()
    stage129_report_path = stage129_report or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_retrieval_integration_validation_stage129.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_integration_failure_review_stage130.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_integration_failure_review_stage130_visuals"
    )

    report = review_primeqa_hybrid_agent_integration_failure_patterns(
        stage129_report_path=stage129_report_path,
        user_confirmed_review=user_confirmed_review,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_agent_integration_failure_review_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage130 review: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "review_id": report["review_id"],
        "train_cv_failure_review": {
            "deltas": report["train_cv_failure_review"][
                "candidate_vs_control_deltas"
            ],
            "changed_verified_answer_rate": report["train_cv_failure_review"][
                "changed_verified_answer_rate_vs_control"
            ],
        },
        "failure_patterns": report["failure_patterns"],
        "action_boundary": report["action_boundary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
