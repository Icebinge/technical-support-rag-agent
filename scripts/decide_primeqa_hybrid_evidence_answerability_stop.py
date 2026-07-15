from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_stop_decision import (
    decide_primeqa_hybrid_evidence_answerability_stop,
    write_primeqa_hybrid_evidence_answerability_stop_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Write Stage106 PrimeQA hybrid evidence-answerability stop decision."
)


@app.command()
def main(
    stage104_protocol: Annotated[
        Path | None,
        typer.Option("--stage104-protocol", help="Stage104 frozen protocol JSON."),
    ] = None,
    stage105_report: Annotated[
        Path | None,
        typer.Option("--stage105-report", help="Stage105 comparison report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage106 stop decision JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_stop: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stop/--no-user-confirmed-stop",
            help="Required confirmation for stopping this candidate family.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage106 stop/redesign decision from public-safe Stage104/105 reports."""

    settings = ProjectSettings()
    stage104_protocol_path = stage104_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.json"
    )
    stage105_report_path = stage105_report or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_stage105.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_stop_decision_stage106.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals"
    )

    report = decide_primeqa_hybrid_evidence_answerability_stop(
        stage104_protocol_path=stage104_protocol_path,
        stage105_report_path=stage105_report_path,
        user_confirmed_stop=user_confirmed_stop,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_evidence_answerability_stop_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid evidence-answerability stop decision: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stopped_family": {
            "family_id": report["stopped_family"]["family_id"],
            "protocol_id": report["stopped_family"]["protocol_id"],
            "stage105_summary": report["stopped_family"]["stage105_summary"],
            "dev_better_nonselectable_configs": report["stopped_family"][
                "dev_better_nonselectable_configs"
            ],
            "stop_reason": report["stopped_family"]["stop_reason"],
        },
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
