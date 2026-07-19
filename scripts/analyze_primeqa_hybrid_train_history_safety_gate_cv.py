from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_train_history_safety_gate_cv as analysis
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage166 train-only history safety-gate outer CV.")


@app.command()
def main(
    correction_report: Annotated[
        Path | None,
        typer.Option("--correction-report", help="Completed Stage165 correction report."),
    ] = None,
    private_report: Annotated[
        Path | None,
        typer.Option("--private-report", help="Immutable Stage165 private observations."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage166 public report."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output Stage166 SVG directory."),
    ] = None,
    user_confirmed_stage166: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stage166/--no-user-confirmed-stage166",
            help="Required confirmation that the user requested the next stage.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Run strict outer CV without loading development or test."""

    artifact_dir = ProjectSettings().artifact_dir
    report = analysis.run_stage166_safety_gate_outer_cv(
        correction_report_path=correction_report
        or artifact_dir
        / "primeqa_hybrid_train_history_isolation_transition_correction_stage165.json",
        private_report_path=private_report
        or artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165_private.json",
        user_confirmed_stage166=user_confirmed_stage166,
        confirmation_note=confirmation_note,
    )
    visual_dir = visualization_dir or (
        artifact_dir / "primeqa_hybrid_train_history_safety_gate_cv_stage166_visuals"
    )
    visualizations = analysis.write_stage166_visualizations(
        report=report,
        output_dir=visual_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_train_history_safety_gate_cv_stage166.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage166 safety-gate CV: {output_path}")
    if report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "feature_contract": report["feature_contract"],
        "case_summary": report["case_summary"],
        "candidate_family": report["candidate_family"],
        "outer_cv": report["outer_cv"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
