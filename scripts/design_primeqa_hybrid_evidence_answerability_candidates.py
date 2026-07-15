from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_candidate_protocol import (
    freeze_primeqa_hybrid_evidence_answerability_candidate_protocol,
    write_primeqa_hybrid_evidence_answerability_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Freeze Stage103 PrimeQA hybrid evidence-answerability candidate design."
)


@app.command()
def main(
    stage102_report: Annotated[
        Path | None,
        typer.Option("--stage102-report", help="Stage102 decomposition JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage103 protocol report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-protocol/--no-user-confirmed-protocol",
            help="Required confirmation for the Stage103 protocol freeze.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage103 public-safe candidate design protocol report."""

    settings = ProjectSettings()
    stage102_report_path = stage102_report or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals"
    )
    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_report_path,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_evidence_answerability_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage103 candidate protocol: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    protocol = report["frozen_candidate_protocol"]
    return {
        "stage": report["stage"],
        "design_id": report["design_id"],
        "user_confirmation": report["user_confirmation"],
        "stage102_summary": {
            "decision_status": report["stage102_summary"]["decision_status"],
            "recommended_next_direction": report["stage102_summary"][
                "recommended_next_direction"
            ],
            "train_top_bucket": report["stage102_summary"]["train_top_bucket"],
            "dev_top_bucket": report["stage102_summary"]["dev_top_bucket"],
            "train_verified_average_token_f1": report["stage102_summary"][
                "train_verified_average_token_f1"
            ],
            "dev_verified_average_token_f1": report["stage102_summary"][
                "dev_verified_average_token_f1"
            ],
        },
        "bottleneck_summary": {
            "primary_bottleneck_bucket_ids": report["bottleneck_summary"][
                "primary_bottleneck_bucket_ids"
            ],
            "bucket_rows": report["bottleneck_summary"]["bucket_rows"],
        },
        "recommended_execution_order": protocol["recommended_execution_order"],
        "candidate_summary": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "risk_level": candidate["risk_level"],
                "target_buckets": candidate["target_buckets"],
                "target_combined_case_count": candidate[
                    "target_combined_case_count"
                ],
                "priority_score": candidate["priority_score"],
            }
            for candidate in protocol["candidate_policies"]
        ],
        "stage104_contract": protocol["stage104_train_dev_comparison_contract"],
        "blocked_items": protocol["blocked_items"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
