from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_selective_dense_sparse_protocol import (
    freeze_primeqa_hybrid_selective_dense_sparse_protocol,
    write_primeqa_hybrid_selective_dense_sparse_protocol_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze Stage97 PrimeQA hybrid selective dense+sparse protocol.")


@app.command()
def main(
    stage84_report: Annotated[
        Path | None,
        typer.Option("--stage84-report", help="Stage84 second-wave design JSON."),
    ] = None,
    stage96_report: Annotated[
        Path | None,
        typer.Option("--stage96-report", help="Stage96 stop decision JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 dense+sparse feasibility JSON."),
    ] = None,
    stage81_report: Annotated[
        Path | None,
        typer.Option("--stage81-report", help="Stage81 dense+sparse comparison JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage97 protocol report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_candidate: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-candidate/--no-user-confirmed-candidate",
            help="Required confirmation for the Stage96 next candidate.",
        ),
    ] = False,
    confirmed_candidate_id: Annotated[
        str,
        typer.Option("--confirmed-candidate-id", help="Confirmed Stage96 candidate id."),
    ] = "selective_dense_sparse_low_overlap_gate_design",
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage97 selective dense+sparse protocol freeze report."""

    settings = ProjectSettings()
    stage84_report_path = stage84_report or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json"
    )
    stage96_report_path = stage96_report or (
        settings.artifact_dir
        / "primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json"
    )
    stage80_report_path = stage80_report or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
    )
    stage81_report_path = stage81_report or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_selective_dense_sparse_protocol_stage97.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals"
    )

    report = freeze_primeqa_hybrid_selective_dense_sparse_protocol(
        stage84_report_path=stage84_report_path,
        stage96_report_path=stage96_report_path,
        stage80_report_path=stage80_report_path,
        stage81_report_path=stage81_report_path,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_selective_dense_sparse_protocol_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid selective dense+sparse protocol report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "protocol_id": frozen["protocol_id"],
        "candidate_id": frozen["candidate_id"],
        "candidate_policy_grid": frozen["candidate_policy_grid"],
        "dense_cache_contract": frozen["dense_cache_contract"],
        "train_selection_rule": frozen["train_selection_rule"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
