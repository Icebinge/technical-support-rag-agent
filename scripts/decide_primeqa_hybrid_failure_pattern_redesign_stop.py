from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_stop_decision import (
    decide_primeqa_hybrid_failure_pattern_redesign_stop,
    write_primeqa_hybrid_failure_pattern_redesign_stop_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Write Stage110 PrimeQA hybrid failure-pattern redesign stop decision."
)


@app.command()
def main(
    stage109_report: Annotated[
        Path | None,
        typer.Option("--stage109-report", help="Stage109 comparison report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage110 stop decision JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_stop: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stop/--no-user-confirmed-stop",
            help="Required confirmation for stopping this redesign family.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage110 stop decision from the public-safe Stage109 report."""

    settings = ProjectSettings()
    stage109_report_path = stage109_report or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_comparison_stage109.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals"
    )

    report = decide_primeqa_hybrid_failure_pattern_redesign_stop(
        stage109_report_path=stage109_report_path,
        user_confirmed_stop=user_confirmed_stop,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_failure_pattern_redesign_stop_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage110 stop decision: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stopped_family": {
            "family_id": report["stopped_family"]["family_id"],
            "source_protocol_id": report["stopped_family"]["source_protocol_id"],
            "source_analysis_id": report["stopped_family"]["source_analysis_id"],
            "stage109_summary": report["stopped_family"]["stage109_summary"],
            "candidate_family_summary": report["stopped_family"][
                "candidate_family_summary"
            ],
            "dev_improved_train_cv_nonselectable_configs": report["stopped_family"][
                "dev_improved_train_cv_nonselectable_configs"
            ],
            "noop_blocked_configs": report["stopped_family"]["noop_blocked_configs"],
            "stop_reason": report["stopped_family"]["stop_reason"],
        },
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
