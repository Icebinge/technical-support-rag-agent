from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison_protocol import (
    freeze_primeqa_hybrid_evidence_answerability_comparison_protocol,
    write_primeqa_hybrid_evidence_answerability_comparison_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage104 PrimeQA hybrid evidence-answerability comparison protocol."
)


@app.command()
def main(
    stage103_protocol: Annotated[
        Path | None,
        typer.Option("--stage103-protocol", help="Stage103 candidate protocol JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage104 protocol report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage104 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage104 comparison-grid protocol freeze report."""

    settings = ProjectSettings()
    stage103_protocol_path = stage103_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals"
    )
    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_protocol_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_evidence_answerability_comparison_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage104 comparison protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "stage103_summary": {
            "decision_status": report["stage103_summary"]["decision_status"],
            "recommended_direction": report["stage103_summary"][
                "recommended_direction"
            ],
            "recommended_execution_order": report["stage103_summary"][
                "recommended_execution_order"
            ],
        },
        "baseline_reference": frozen["baseline_reference"],
        "candidate_config_grid": frozen["candidate_config_grid"],
        "train_selection_rule": frozen["train_selection_rule"],
        "dev_validation_rule": frozen["dev_validation_rule"],
        "metric_contract": frozen["metric_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
