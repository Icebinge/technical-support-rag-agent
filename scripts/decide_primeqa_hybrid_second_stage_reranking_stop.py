from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_stop_decision import (
    decide_primeqa_hybrid_second_stage_reranking_stop,
    write_primeqa_hybrid_second_stage_reranking_stop_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Write Stage119 PrimeQA hybrid second-stage reranking stop decision."
)


@app.command()
def main(
    stage118_report: Annotated[
        Path | None,
        typer.Option("--stage118-report", help="Stage118 validation report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage119 stop decision JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_stop: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stop/--no-user-confirmed-stop",
            help="Required confirmation for stopping this reranking family.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage119 stop decision from the public-safe Stage118 report."""

    settings = ProjectSettings()
    stage118_report_path = stage118_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_stage_reranking_validation_stage118.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_second_stage_reranking_stop_decision_stage119.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals"
    )

    report = decide_primeqa_hybrid_second_stage_reranking_stop(
        stage118_report_path=stage118_report_path,
        user_confirmed_stop=user_confirmed_stop,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_second_stage_reranking_stop_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage119 stop decision: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    stopped = report["stopped_family"]
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stopped_family": {
            "family_id": stopped["family_id"],
            "source_protocol_id": stopped["source_protocol_id"],
            "source_analysis_id": stopped["source_analysis_id"],
            "stage118_summary": stopped["stage118_summary"],
            "candidate_family_summary": stopped["candidate_family_summary"],
            "train_cv_positive_signal_but_blocked_configs": stopped[
                "train_cv_positive_signal_but_blocked_configs"
            ],
            "dev_report_observations": stopped["dev_report_observations"],
            "stop_reason": stopped["stop_reason"],
        },
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
